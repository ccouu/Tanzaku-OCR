"""Local canvas OBB editor. Only explicit human saves create reviewed labels."""
import argparse
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from datetime import datetime, timezone
from urllib.parse import urlparse

from src.common import digest
from src.common.paths import resolve_path
from ..common import DEFAULT_ROOT, read_json, save_pair, validate_shapes


class ReviewStore:
    def __init__(self, root):
        self.root = resolve_path(root).resolve()
        self.records = {r['id']: r for r in read_json(self.root / 'manifest.json')['records']}
        self.lock = threading.Lock()

    def load(self, ident):
        record = self.records[ident]
        reviewed = self.root / 'labels' / (ident + '.json')
        path = reviewed if reviewed.exists() else self.root / 'predictions' / (ident + '.json')
        doc = read_json(path)
        if doc['image_sha256'] != record['sha256']:
            raise ValueError('Annotation image hash mismatch')
        validate_shapes(doc['shapes'], record['width'], record['height'])
        return dict(record=record, annotation=doc, revision=digest(path))

    def save(self, ident, payload):
        with self.lock:
            current = self.load(ident)
            if payload.get('revision') != current['revision']:
                raise ValueError('Annotation changed elsewhere; reload before saving')
            record = current['record']
            if record['role'] != 'page':
                raise ValueError('Reference images and CTC crops cannot become page training labels')
            shapes = validate_shapes(payload['shapes'], record['width'], record['height'])
            if not shapes and payload.get('confirm_empty') is not True:
                raise ValueError('Explicit confirmation required for an empty negative image')
            if digest(self.root / 'images' / record['image']) != record['sha256']:
                raise ValueError('Image checksum changed')
            doc = copy.deepcopy(current['annotation'])
            doc['shapes'] = shapes
            doc['flags'] = {'reviewed': True, 'confirmed_empty': not shapes}
            doc['reviewed_at_utc'] = datetime.now(timezone.utc).isoformat()
            doc['review_tool'] = 'stage2-local-canvas'
            doc['source_info_path'] = '../provenance/' + ident + '.json'
            for shape in shapes:
                shape['shape_type'] = 'polygon'
                shape.setdefault('flags', {})['reviewed'] = True
            save_pair(self.root / 'labels', record, doc)
            return self.load(ident)


def make_server(root, port=7860):
    store = ReviewStore(root)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data, content_type='application/json', status=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def do_GET(self):
            if not self.valid_host():
                return self.respond({'error': 'Invalid host'}, status=403)
            path = urlparse(self.path).path
            try:
                if path == '/':
                    html = Path(__file__).with_name('review.html').read_text(encoding='utf-8').replace('__TOKEN__', token)
                    return self.respond(html.encode(), 'text/html; charset=utf-8')
                if path == '/api/images':
                    return self.respond([dict(id=r['id'], image=r['image'], role=r['role'], source_path=r['source_path'],
                        predicted=(store.root/'predictions'/(r['id']+'.json')).exists(),
                        reviewed=(store.root/'labels'/(r['id']+'.json')).exists()) for r in store.records.values()])
                ident = path.rsplit('/', 1)[-1]
                record = store.records[ident]
                if path.startswith('/api/annotation/'):
                    return self.respond(store.load(ident))
                if path.startswith('/image/'):
                    image = store.root / 'images' / record['image']
                    import mimetypes
                    return self.respond(image.read_bytes(), mimetypes.guess_type(image)[0] or 'application/octet-stream')
            except (KeyError, FileNotFoundError):
                return self.respond({'error': 'Image or prediction not found'}, status=404)
            except ValueError as exc:
                return self.respond({'error': str(exc)}, status=400)
            self.respond({'error': 'Not found'}, status=404)

        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-Review-Token') != token:
                return self.respond({'error': 'Invalid local review token'}, status=403)
            try:
                if not self.path.startswith('/api/annotation/'):
                    return self.respond({'error': 'Not found'}, status=404)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4000000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                result = store.save(self.path.rsplit('/', 1)[-1], payload)
                self.respond(result)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond({'error': str(exc)}, status=400)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    server = make_server(args.root, args.port)
    print(f'Review UI: http://localhost:{server.server_port} (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
