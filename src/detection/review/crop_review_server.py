"""Loopback-only before/after crop inspection and explicit user feedback."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from urllib.parse import unquote,urlparse
from src.common import dump_json


def make_server(root,port):
    root=Path(root).resolve();index=json.loads((root/'index.json').read_text(encoding='utf-8'));ids={r['id'] for r in index['menus']};token=secrets.token_urlsafe(32);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,data,mime='application/json',status=200):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def host_ok(self):return self.headers.get('Host') in (f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}')
        def do_GET(self):
            if not self.host_ok():return self.send({'error':'InvalidHost'},status=403)
            path=unquote(urlparse(self.path).path)
            if path=='/':return self.send(Path(__file__).with_name('crop_review.html').read_text(encoding='utf-8').replace('__TOKEN__',token).encode(),'text/html; charset=utf-8')
            if path=='/api/index':return self.send(index)
            if path=='/api/feedback':return self.send(json.loads((root/'feedback.json').read_text()) if (root/'feedback.json').exists() else {})
            if path.startswith('/api/menu/'):
                ident=path.rsplit('/',1)[-1]
                if ident not in ids:return self.send({'error':'UnknownMenu'},status=404)
                return self.send((root/'menus'/(ident+'.json')).read_bytes())
            if path.startswith('/assets/'):
                target=(root/path[len('/assets/'):]).resolve()
                if not target.is_relative_to(root) or not target.is_file():return self.send({'error':'MissingAsset'},status=404)
                return self.send(target.read_bytes(),mimetypes.guess_type(target)[0] or 'application/octet-stream')
            self.send({'error':'NotFound'},status=404)
        def do_POST(self):
            if not self.host_ok() or self.path!='/api/feedback' or self.headers.get('X-Review-Token')!=token:return self.send({'error':'Forbidden'},status=403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=100000:raise ValueError('Invalidsize')
                body=json.loads(self.rfile.read(size));ident=body['id'];status=body['status'];note=body.get('note','');crop=body.get('crop');variant=body.get('variant','conservative')
                if ident not in ids or status not in ('accepted','needs_fix','unreviewed') or not isinstance(note,str) or len(note)>5000:raise ValueError('Invalidfeedback')
                if crop is not None and (not isinstance(crop,str) or len(crop)>100):raise ValueError('Invalidcrop')
                if variant not in ('conservative','adjacent'):raise ValueError('Invalidvariant')
                with lock:
                    file=root/'feedback.json';data=json.loads(file.read_text()) if file.exists() else {}
                    from datetime import datetime,timezone
                    event=dict(status=status,note=note,crop=crop,variant=variant,updated_utc=datetime.now(timezone.utc).isoformat())
                    history=data.get(ident,{}).get('history',[])+[event]
                    data[ident]=dict(event,history=history[-2000:]);dump_json(file,data)
                self.send(data[ident])
            except (ValueError,KeyError,TypeError) as exc:self.send({'error':str(exc)},status=400)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=7864);a=p.parse_args()
    server=make_server(a.root,a.port);print(f'Crop review: http://127.0.0.1:{server.server_port}',flush=True);server.serve_forever()
