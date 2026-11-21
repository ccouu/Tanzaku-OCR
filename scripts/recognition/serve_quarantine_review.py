"""Local read-only review of the Tabelog pilot's OOV and overlap exclusions."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import mimetypes
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / 'data/raw/tabelog_selected_menu_manual/16112026'
DEFAULT_MANIFEST = ROOT / 'outputs/recognition/experiments/l2_dishlex_tabelog30_yolo20_synth50_1000/manifests/new_tabelog_train_quarantine.jsonl'
DEFAULT_OUTPUT = ROOT / 'outputs/recognition/review/tabelog_16112026_quarantine'
REASONS = {'oov_1779', 'core_overlap_review_required'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_review(manifest, output):
    """Read original records, retaining their stable IDs and every exclusion flag."""
    manifest = manifest.resolve()
    output = output.resolve()
    if not manifest.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError('Manifest and output must stay in workspace')
    crop_manifest = BATCH / 'yolo_crops/crops_manifest.json'
    crops = read_json(crop_manifest)['crops']
    by_id = {c['crop_id']: c for c in crops}
    rows = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
    rows = [r for r in rows if r['quarantine_reason'] in REASONS]
    assets, records = {}, []
    for i, row in enumerate(rows):
        crop = by_id[row['raw_crop_id']]
        crop_path = (BATCH / 'yolo_crops' / crop['display_crop']).resolve()
        menu_path = (BATCH / crop['source_image']).resolve()
        for path in (crop_path, menu_path):
            if not path.is_relative_to(BATCH) or not path.is_file():
                raise ValueError(f'Missing or out-of-batch image: {path}')
        digest = hashlib.sha256(crop_path.read_bytes()).hexdigest()
        if digest != row['display_sha256']:
            raise ValueError(f'Crop hash mismatch: {row["crop_id"]}')
        assets[('crop', i)] = crop_path
        assets[('menu', i)] = menu_path
        overlap_ids = set(crop.get('overlap_source_ids', []))
        neighbors = [c for c in crops if c['source_menu'] == crop['source_menu']
                     and c['crop_id'] != crop['crop_id']
                     and overlap_ids.intersection(c.get('source_ids', []))]
        with Image.open(menu_path) as image:
            size = list(image.size)
        records.append({**row, 'index': i, 'source_image': crop['source_image'],
                        'menu_size': size, 'crop_polygon': crop['padded_polygon'],
                        'core_polygon': crop['obb_polygon'],
                        'neighbors': [{'crop_id': c['crop_id'], 'polygon': c['obb_polygon']} for c in neighbors],
                        'crop_url': f'/asset/crop/{i}', 'menu_url': f'/asset/menu/{i}'})
    data = {'records': records, 'counts': dict(Counter(r['quarantine_reason'] for r in rows)),
            'menus': len({r['menu_key'] for r in rows}),
            'manifest': str(manifest.relative_to(ROOT)),
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'note': 'Training-split primary exclusion groups only. Flags can overlap; VLM labels are provisional. Review exports do not modify training data.'}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'review_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'index.html').write_text(HTML, encoding='utf-8')
    return data, assets


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Tabelog crop quarantine review</title><style>
body{font:15px system-ui;margin:0;background:#101722;color:#e7edf5}header{padding:18px;position:sticky;top:0;background:#172232;z-index:2;border-bottom:1px solid #506070}
h1{font-size:22px;margin:0 0 8px}button,select,input,textarea{font:inherit;padding:7px;background:#24354b;color:#fff;border:1px solid #60738b;border-radius:5px}button{cursor:pointer}label{display:inline-block;margin:5px}main{padding:18px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}.card{padding:14px;background:#1d2a3b;border:1px solid #53647b;border-radius:8px;overflow-wrap:anywhere}.card img{width:100%;height:190px;object-fit:contain;background:#fff;cursor:zoom-in}.text{font-size:23px;white-space:pre-wrap;margin:12px 0}mark{background:#ffb857;color:#171717;border-radius:2px}small,.meta{color:#bac9dd}.tags{color:#ffc16a}textarea{box-sizing:border-box;width:100%;min-height:65px;margin-top:7px}#pages{display:flex;gap:12px;padding:14px 0}dialog{width:94vw;max-width:1600px;max-height:94vh;background:#152033;color:#fff;border:1px solid #9babc0}dialog::backdrop{background:#000b}.context{overflow:auto;max-height:75vh}.stage{position:relative;min-width:100%}.stage img{display:block;width:100%;height:auto}.stage svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}#status{margin:8px 0}.primary{color:#ffce83}.card input{box-sizing:border-box;width:100%;margin-top:6px}
</style><header><h1>Tabelog: 280 OOV + 214 unresolved-overlap crops</h1>
<div id="status">Loading…</div><div>
<label>Group <select id="group"><option value="oov_1779">OOV — 280</option><option value="core_overlap_review_required">Unresolved overlap — 214</option><option value="all">Both — 494</option></select></label>
<label>Menu <select id="menu"><option value="all">All menus</option></select></label>
<label>Search text / missing char <input id="search" placeholder="鱸, crop ID, text…"></label>
<label>Review <select id="filter"><option value="all">All</option><option value="unreviewed">Unreviewed</option><option value="reviewed">Reviewed</option></select></label>
<button id="export">Export ALL reviews</button><button id="importBtn">Import reviews</button><input type="file" id="importFile" accept="application/json" hidden>
</div><small>OOV = characters absent from the current 1,779-character vocabulary. Overlap = detected text cores still intersect; no strokes were clipped to force separation. These are primary exclusion categories, not mutually exclusive visual problems. Labels are provisional VLM text.</small></header>
<main><div id="pages"><button id="prev">Previous</button><span id="pageLabel"></span><button id="next">Next</button></div><div id="grid" class="grid"></div></main>
<dialog id="modal"><button id="close">Close</button> <button id="zoom">Toggle zoom</button><p id="caption"></p><div class="context"><div id="stage" class="stage"></div></div></dialog>
<script>
const el=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';let data, reviews={}, page=0, key;
const mk=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e};
function persist(){localStorage.setItem(key,JSON.stringify(reviews));}
function highlighted(text,chars){const e=mk('div',undefined,'text');const missing=new Set(chars||[]);for(const c of text)e.appendChild(mk(missing.has(c)?'mark':'span',c));return e;}
function changed(r,field,value){reviews[r.crop_id]={...(reviews[r.crop_id]||{}),[field]:value,updated_at:new Date().toISOString()};persist();}
function visible(){const q=el('search').value.trim();return data.records.filter(r=>(el('group').value==='all'||r.quarantine_reason===el('group').value)&&(el('menu').value==='all'||r.menu_key===el('menu').value)&&(!q||[r.original_text,r.canonical_text,r.crop_id,...r.oov_chars].join(' ').includes(q))&&(el('filter').value==='all'||(el('filter').value==='reviewed')===Boolean(reviews[r.crop_id]?.decision)));}
function context(r,full){const s=el('stage');s.replaceChildren();s.style.width='100%';const im=mk('img');im.src=full?r.menu_url:r.crop_url;im.alt=r.crop_id;s.appendChild(im);el('caption').textContent=full?`${r.source_image} — cyan: selected crop; orange: selected core; red: intersecting neighbors (${r.neighbors.length}). Coordinates from the original crop manifest.`:r.crop_id;
if(full){const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${r.menu_size[0]} ${r.menu_size[1]}`);for(const item of [{polygon:r.crop_polygon,color:'#00e5ff'},{polygon:r.core_polygon,color:'#ffb13c'},...r.neighbors.map(n=>({...n,color:'#ff4141'}))]){const p=document.createElementNS(ns,'polygon');p.setAttribute('points',item.polygon.map(p=>p.join(',')).join(' '));p.setAttribute('fill','none');p.setAttribute('stroke',item.color);p.setAttribute('stroke-width','3');p.setAttribute('vector-effect','non-scaling-stroke');svg.appendChild(p);}s.appendChild(svg);}el('modal').showModal();}
function render(){const rows=visible(), max=Math.max(1,Math.ceil(rows.length/24));page=Math.max(0,Math.min(page,max-1));el('grid').replaceChildren();el('pageLabel').textContent=`${rows.length} crops · page ${page+1}/${max}`;el('prev').disabled=page===0;el('next').disabled=page>=max-1;
for(const r of rows.slice(page*24,(page+1)*24)){const c=mk('article',undefined,'card'), rev=reviews[r.crop_id]||{};c.appendChild(mk('b',r.quarantine_reason==='oov_1779'?'OOV':'Unresolved overlap','primary'));c.appendChild(mk('p',r.raw_crop_id,'meta'));const img=mk('img');img.src=r.crop_url;img.loading='lazy';img.alt=r.original_text;img.onclick=()=>context(r,false);c.appendChild(img);c.appendChild(highlighted(r.original_text,r.oov_chars));
if(r.canonical_text!==r.original_text)c.appendChild(mk('div',`Canonical: ${r.canonical_text}`,'meta'));
c.appendChild(mk('div',r.oov_chars.length?`Missing: ${r.oov_chars.map(ch=>`${ch} U+${ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')}`).join(' · ')}`:'No OOV characters','tags'));
c.appendChild(mk('p',`Status ${r.status} · ${r.orientation_flag?'Vertical':'Horizontal'} · native H64 width ${r.native_w64}px · ${r.review_required?'overlap flagged':'no overlap flag'} · ${r.over_512?'over512':'fits512'}`,'meta'));
const b=mk('button','Show source menu + overlap');b.onclick=()=>context(r,true);c.appendChild(b);const sel=mk('select');for(const [value,label] of [['','Unreviewed'],['label_ok','Label correct'],['label_fix','Label needs correction'],['boundary_fix','Crop boundary needs repair'],['exclude','Exclude / no useful text'],['uncertain','Uncertain']]){const o=mk('option',label);o.value=value;sel.appendChild(o);}sel.value=rev.decision||'';sel.onchange=()=>changed(r,'decision',sel.value);c.appendChild(sel);
const corrected=mk('input');corrected.placeholder='Proposed corrected transcription (optional)';corrected.value=rev.corrected_text||'';corrected.oninput=()=>changed(r,'corrected_text',corrected.value);c.appendChild(corrected);const note=mk('textarea');note.placeholder='Notes: glyph, duplicate line, clipping…';note.value=rev.notes||'';note.oninput=()=>changed(r,'notes',note.value);c.appendChild(note);el('grid').appendChild(c);}}
for(const id of ['group','menu','filter'])el(id).onchange=()=>{page=0;render();};el('search').oninput=()=>{page=0;render();};el('prev').onclick=()=>{page--;render();};el('next').onclick=()=>{page++;render();};el('close').onclick=()=>el('modal').close();el('zoom').onclick=()=>{el('stage').style.width=el('stage').style.width==='200%'?'100%':'200%';};
el('export').onclick=()=>{const payload={schema_version:1,manifest_sha256:data.manifest_sha256,exported_at:new Date().toISOString(),records:data.records.map(r=>({crop_id:r.crop_id,source_menu:r.source_menu,source_image_sha256:r.source_image_sha256,display_sha256:r.display_sha256,original_text:r.original_text,canonical_text:r.canonical_text,oov_chars:r.oov_chars,quarantine_reason:r.quarantine_reason,review:reviews[r.crop_id]||{}}))};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=mk('a');a.href=url;a.download='tabelog_16112026_quarantine_reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
el('importBtn').onclick=()=>el('importFile').click();el('importFile').onchange=async e=>{try{const obj=JSON.parse(await e.target.files[0].text());if(obj.manifest_sha256!==data.manifest_sha256)throw Error('Manifest hash mismatch');const ids=new Set(data.records.map(r=>r.crop_id));for(const r of obj.records||[])if(ids.has(r.crop_id)&&r.review)reviews[r.crop_id]=r.review;persist();render();}catch(err){alert(err.message);}};
fetch('/api/data').then(r=>r.json()).then(d=>{data=d;key='tabelog-quarantine-'+d.manifest_sha256;try{reviews=JSON.parse(localStorage.getItem(key)||'{}');}catch{}const menus=[...new Map(d.records.map(r=>[r.menu_key,r.source_menu])).entries()];for(const [value,label] of menus){const o=mk('option',label);o.value=value;el('menu').appendChild(o);}el('status').textContent=`${d.records.length} crops · ${d.counts.oov_1779} OOV · ${d.counts.core_overlap_review_required} overlap · ${d.menus} training menus. Changes save only in this browser; export JSON to share.`;render();}).catch(e=>el('status').textContent=e.message);
</script></html>'''


def make_handler(data, assets, output):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                payload, mime = HTML.encode('utf-8'), 'text/html; charset=utf-8'
            elif path in ('/api/data', '/api/health'):
                value = data if path == '/api/data' else {'status': 'ok', 'counts': data['counts'], 'total': len(data['records'])}
                payload, mime = json.dumps(value, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8'
            else:
                match = re.fullmatch(r'/asset/(crop|menu)/(\d+)', path)
                if not match or (match[1], int(match[2])) not in assets:
                    self.send_error(404)
                    return
                file = assets[(match[1], int(match[2]))]
                if match[1] == 'menu':
                    # Strip EXIF metadata without transposing: polygons refer to
                    # raw pixels from the historical batch exporter.
                    with Image.open(file) as image:
                        image = image.convert('RGB')
                        image.thumbnail((1800, 1800))
                        buf = io.BytesIO()
                        image.save(buf, format='JPEG', quality=88)
                        payload, mime = buf.getvalue(), 'image/jpeg'
                else:
                    payload, mime = file.read_bytes(), mimetypes.guess_type(file.name)[0] or 'image/png'
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    data, assets = build_review(args.manifest, args.output)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(data, assets, args.output))
    runtime = {'url': f'http://127.0.0.1:{server.server_port}/', 'counts': data['counts']}
    (args.output / 'runtime.json').write_text(json.dumps(runtime), encoding='utf-8')
    print(json.dumps(runtime), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
