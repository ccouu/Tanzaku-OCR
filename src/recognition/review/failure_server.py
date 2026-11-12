"""Loopback versioned failure-review UI; allowlisted assets and separate reviews."""
import argparse
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
import re
from pathlib import Path
import secrets
import threading
from urllib.parse import urlparse,unquote

CATEGORIES={'label_correct','wrong_label','clipped','overlap','no_text','uncertain','unreviewed'}
PROJECT=Path(__file__).resolve().parents[3]


def make_server(root,port,workspace=PROJECT):
    root=Path(root).resolve();workspace=Path(workspace).resolve()
    index=json.loads((root/'index.json').read_text(encoding='utf-8'))
    version=re.sub(r'[^a-z0-9_-]','',index.get('metadata',{}).get('version','V38').lower())
    rows={r['id']:r for r in index['rows']};vocab=set(index.get('vocab_chars',[]))
    manifest=json.loads((root/'assets.json').read_text(encoding='utf-8'))
    allowed=[(workspace/p).resolve() for p in manifest['allowed_roots']]
    assert all(p.is_relative_to(workspace) for p in allowed)
    assets={k:(workspace/v).resolve() for k,v in manifest['assets'].items()}
    assert all(any(p.is_relative_to(folder) for folder in allowed) for p in assets.values())
    token=secrets.token_urlsafe(32);lock=threading.Lock()
    reviews_path=root/'reviews.json'
    def reviews():return json.loads(reviews_path.read_text(encoding='utf-8')) if reviews_path.exists() else {}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,data,mime='application/json; charset=utf-8',status=200,download=False):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; object-src 'none'; frame-ancestors 'none'")
            if download:self.send_header('Content-Disposition',f'attachment; filename="{version}_training_reviews.jsonl"')
            self.end_headers();self.wfile.write(data)
        def host_ok(self):return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        def do_GET(self):
            if not self.host_ok():return self.send({'error':'Invalid host'},status=403)
            path=unquote(urlparse(self.path).path)
            if path=='/':return self.send(Path(__file__).with_name('failure_review.html').read_text(encoding='utf-8').replace('__TOKEN__',token).encode('utf-8'),'text/html; charset=utf-8')
            if path=='/api/index':return self.send(index)
            if path=='/api/reviews':
                with lock:return self.send(reviews())
            if path=='/api/export':
                with lock:data=reviews()
                exports=[dict(crop_id=k,source_menu=rows[k]['source_menu'],original_label=rows[k]['label'],prediction=rows[k]['prediction'],
                              input_sha256=rows[k]['input_sha256'],review=v) for k,v in data.items() if k in rows and v['category']!='unreviewed']
                return self.send(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exports).encode('utf-8'),'application/x-ndjson; charset=utf-8',download=True)
            if path.startswith('/assets/'):
                key=path[len('/assets/'):];target=assets.get(key)
                if target is None or not target.is_file() or not any(target.resolve().is_relative_to(p) for p in allowed):return self.send({'error':'Unknown asset'},status=404)
                return self.send(target.read_bytes(),mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            self.send({'error':'Not found'},status=404)
        def do_POST(self):
            if not self.host_ok() or self.path!='/api/review' or self.headers.get('X-Review-Token')!=token:
                return self.send({'error':'Forbidden'},status=403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=20000:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(size));ident=body['id'];category=body['category']
                correction=body.get('correction','').strip();note=body.get('note','').strip()
                if ident not in rows or category not in CATEGORIES:raise ValueError('Unknown crop or category')
                if not isinstance(correction,str) or not isinstance(note,str) or len(correction)>256 or len(note)>4000:
                    raise ValueError('Invalid correction or note')
                if any(c in correction for c in '\n\r\t'):raise ValueError('Correction must be one line')
                if category=='wrong_label' and (not correction or correction==rows[ident]['label']):
                    raise ValueError('Enter a different corrected transcription for a wrong label')
                if category!='wrong_label':correction=''
                entry=dict(category=category,correction=correction,note=note,
                           correction_oov=sorted(set(correction)-vocab),updated_utc=datetime.now(timezone.utc).isoformat())
                with lock:
                    data=reviews();old=data.get(ident)
                    history=(old.get('history',[])+[{k:v for k,v in old.items() if k!='history'}]) if old else []
                    entry['history']=history[-100:];data[ident]=entry
                    temporary=reviews_path.with_suffix('.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(reviews_path)
                self.send(entry)
            except (ValueError,KeyError,TypeError,AttributeError) as exc:self.send({'error':str(exc)},status=400)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=7866);args=p.parse_args()
    server=make_server(args.root,args.port);print(f'Failure review: http://127.0.0.1:{server.server_port}/',flush=True);server.serve_forever()
