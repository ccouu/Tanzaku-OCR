"""Build a local opening-parenthesis review from existing V50 predictions."""
import base64
from collections import Counter
from io import BytesIO
from pathlib import Path
import json
import shutil
from PIL import Image, ImageDraw
from src.recognition.label_normalization import normalize_parentheses
from src.recognition.metrics import align_errors
from src.recognition.utils import sha, write_json

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/recognition/pilots/v50_normalized_matched_2000'
INPUT=ROOT/'data/processed/gemini_banana_mobile_menus_50_crops'
OUT=ROOT/'outputs/recognition/review/v50_opening_parentheses'


def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    OUT.mkdir(parents=True,exist_ok=True); assets=OUT/'images'; assets.mkdir(exist_ok=True)
    metadata=[json.loads(s) for s in (ROOT/'outputs/recognition/training/v40_new_characters_2000/yolo_labels.jsonl').read_text(encoding='utf-8').splitlines()]
    menu_root=Path(load(INPUT/'crops_manifest.json')['input_dir'])
    evaluations={
        'V43':load(RUN/'control/initial_fp16.json'),
        'Control selected 1500':load(RUN/'control/step_1500_raw_fp16.json'),
        'Control final 2000':load(RUN/'control/step_2000_raw_fp16.json'),
        'BiLSTM final 2000':load(RUN/'bilstm/step_2000_raw_fp16.json')}
    records=[];menus=set();references={}
    for split,probe in [('val_holdout','yolo_current_holdout'),('train','yolo_current_train')]:
        labels=[r for r in metadata if r['split']==split]
        for i,row in enumerate(labels):
            if '(' not in normalize_parentheses(row['text']): continue
            ident=row['crop_id'];predictions={}
            for name,e in evaluations.items():
                modes={}
                for field in ('validation','text_normalized_validation','canonical_validation'):
                    p=e[field][probe]['0.0']['predictions'][i]
                    assert p['index']==i and p['truth']==(row['text'] if field=='validation' else normalize_parentheses(row['text']))
                    ops=align_errors(p['truth'],p['prediction'])
                    modes[field]=dict(text=p['prediction'],errors=p['error_counts'],
                        opening_hits=sum(op['type']=='match' and op.get('truth')=='(' for op in ops),
                        alignment=ops)
                predictions[name]=modes
            native=INPUT/row['display_crop'];ctc=INPUT/row['image']
            assert native.is_file() and sha(ctc)==row['sha256']
            shutil.copyfile(native,assets/f'{ident}_native.png');shutil.copyfile(ctc,assets/f'{ident}_ctc.png')
            with Image.open(native) as im: native_size=im.size
            menu=menu_root/row['source_image'];assert menu.is_file()
            with Image.open(menu) as im:
                im=im.convert('RGB'); points=row['padded_polygon']; core=row['obb_polygon']
                xs=[x for x,y in points+core];ys=[y for x,y in points+core]
                margin=max(48,int(max(max(xs)-min(xs),max(ys)-min(ys))*.25))
                x0=max(0,int(min(xs))-margin);y0=max(0,int(min(ys))-margin)
                x1=min(im.width,int(max(xs))+margin);y1=min(im.height,int(max(ys))+margin)
                context=im.crop((x0,y0,x1,y1));draw=ImageDraw.Draw(context)
                for polygon,color in [(points,'#28d9ff'),(core,'#ff8b53')]:
                    pp=[(x-x0,y-y0) for x,y in polygon];draw.line(pp+[pp[0]],fill=color,width=2)
                context.thumbnail((1000,700));context.save(assets/f'{ident}_context.jpg',quality=90)
                if row['source_menu'] not in menus:
                    im.thumbnail((1800,1800));im.save(assets/f'{row["source_menu"]}.jpg',quality=90);menus.add(row['source_menu'])
            records.append(dict(id=ident,menu=row['source_menu'],split=split,label=row['text'],
                canonical_label=normalize_parentheses(row['text']),predictions=predictions,
                native_size=native_size,canvas=row['canvas'],valid_width=row['valid_width'],
                vertical=row['is_vertical'],geometry_review_required=row.get('review_required',False),
                overlap_ids=row.get('overlap_source_ids',[]),confidence=row['confidence'],
                input_sha256=row['sha256']))
    records.sort(key=lambda r:(r['split']!='val_holdout',r['id']))
    payload=dict(rows=records,metadata=dict(precision='FP16',new_inference=False,
        counts=dict(Counter(r['split'] for r in records)),labels='VLM-generated, not independently verified',
        checkpoint='BiLSTM final raw 2000 is not selected or promoted; production remains V43'))
    write_json(OUT/'index.json',payload)
    html=HTML.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/'))
    (OUT/'index.html').write_text(html,encoding='utf-8')
    write_json(OUT/'build_audit.json',dict(rows=len(records),counts=payload['metadata']['counts'],
        saved_predictions_only=True,source_images_and_labels_unchanged=True,no_visual_llm_calls=True,
        page_sha256=sha(OUT/'index.html')))
    print(json.dumps(payload['metadata']));print(OUT/'index.html')


HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V50 · Opening parentheses review</title><style>
:root{color-scheme:dark;font-family:system-ui,'Yu Gothic',sans-serif;background:#10151b;color:#e5eaf0}body{margin:0}header{padding:25px 4vw 15px;background:#17212c}h1{font-size:27px;margin:0 0 10px}p{line-height:1.6;max-width:1150px;color:#bac8d7}nav{position:sticky;top:0;padding:12px 4vw;background:#17212cf5;display:flex;gap:10px;flex-wrap:wrap;z-index:2}select,input,button,textarea{background:#253447;color:#fff;border:1px solid #506279;border-radius:6px;padding:8px}button{cursor:pointer}main{padding:16px 4vw}.card{background:#1a2531;padding:18px;border:1px solid #38495d;border-radius:12px;margin-bottom:22px}.title{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.badge{font-size:12px;padding:4px 8px;background:#35465e;border-radius:15px}.hold{background:#573d16}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:16px}img{max-width:100%;object-fit:contain;background:#fff}.native{max-height:140px;image-rendering:auto;min-height:64px}.ctcwrap{position:relative;width:fit-content;max-width:100%}.ctc{display:block;height:64px;width:auto;max-width:100%}.edge{position:absolute;height:100%;top:0;border-left:2px dashed #ff325b;pointer-events:none}.context{max-height:330px}.caption{font-size:12px;color:#b0bdcd;margin:6px 0 10px}.truth{font-size:24px;white-space:pre-wrap;overflow-wrap:anywhere;background:#101820;padding:12px;border-radius:6px}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{padding:10px;text-align:left;border-bottom:1px solid #3a4a5c;overflow-wrap:anywhere}th:first-child{width:29%}.prediction{font-size:20px;white-space:pre-wrap}.wrong{color:#ffac9f}.ok{color:#8ce0ad}.review{margin-top:16px;display:flex;gap:10px;flex-wrap:wrap}.review textarea{flex:1;min-width:220px;min-height:45px}a{color:#7acaff}.small{font-size:13px;color:#b4c2d1}mark{background:#4f4075;color:#fff}summary{cursor:pointer;padding:10px 0}@media(max-width:800px){.grid{grid-template-columns:1fr}header{padding:18px}nav{padding:10px}main{padding:12px}}
</style><header><h1>Opening parentheses · V43 versus V50</h1>
<p>Review the <b>13 held-out lines</b> first. The extra 110 training lines provide comparisons. Labels came from a visual LLM and may be wrong. Check whether an opening bracket is visible, clipped, small or low contrast before blaming recognition. No new model inference or visual-LLM calls were made.</p>
<p class="small">Orange outline = detector box; cyan = crop polygon. Red dashed line on the CTC image = valid-width endpoint. Context stays in source orientation; native crops are upright. BiLSTM final raw 2000 was <b>not selected or promoted</b>. Production remains V43. Once inspected, this holdout is development data, not a fresh blind test.</p></header>
<nav><select id="split"><option value="val_holdout">Held-out menus (13 lines)</option><option value="train">Training menus (110 lines)</option><option value="all">All 123 lines</option></select>
<select id="mode"><option value="canonical_validation">Pooled normalized decoding</option><option value="text_normalized_validation">Normalize after decoding</option><option value="validation">Literal decoding</option></select>
<input id="search" placeholder="Search label / menu / crop"><select id="reviewFilter"><option value="all">All review states</option><option value="pending">Pending review</option></select><button id="export">Export my review JSON</button><span id="count"></span></nav><main id="main"></main>
<script>const DATA=__PAYLOAD__;const KEY='tanzaku-v50-opening-review-v1';let notes={};try{notes=JSON.parse(localStorage.getItem(KEY)||'{}')}catch{};
const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function save(){try{localStorage.setItem(KEY,JSON.stringify(notes))}catch(e){alert('Browser storage unavailable; export your review before closing.')}}
function render(){const split=$('split').value,q=$('search').value.toLowerCase(),mode=$('mode').value;const rr=DATA.rows.filter(r=>(split==='all'||r.split===split)&&(!q||(r.id+' '+r.label).toLowerCase().includes(q))&&($('reviewFilter').value!=='pending'||!notes[r.id]?.cause));$('count').textContent=`${rr.length} lines · ${Object.keys(notes).length} saved reviews`;
$('main').innerHTML=rr.map(r=>{const n=notes[r.id]||{};const label=mode==='validation'?r.label:r.canonical_label;return `<article class="card" id="${r.id}"><div class="title"><b>${r.id}</b><span class="badge ${r.split==='val_holdout'?'hold':''}">${r.split==='val_holdout'?'HELD OUT':'TRAIN'}</span><span class="small">${r.vertical?'Vertical → rotated':'Horizontal'} · native ${r.native_size.join(' × ')} · bucket ${r.canvas} · valid width ${r.valid_width}</span></div>
<div class="grid"><section><div class="caption">Original VLM label — verify visually</div><div class="truth">${esc(r.label)}</div>${r.label!==r.canonical_label?`<div class="caption">Normalized target: ${esc(r.canonical_label)}</div>`:''}<div class="caption">Native upright crop · click to inspect actual pixels</div><a target="_blank" href="images/${r.id}_native.png"><img class="native" src="images/${r.id}_native.png" loading="lazy" alt="Native crop ${r.id}"></a><div class="caption">Actual H64 CTC input · includes bucket padding</div><div class="ctcwrap"><img class="ctc" src="images/${r.id}_ctc.png" loading="lazy" alt="CTC input"><span class="edge" style="left:${100*r.valid_width/r.canvas}%"></span></div>
<table><thead><tr><th>Checkpoint</th><th>Prediction / errors (D I S)</th></tr></thead><tbody>${Object.entries(r.predictions).map(([name,all])=>{const p=all[mode];return `<tr><td>${esc(name)}</td><td><div class="prediction ${p.text===label?'ok':'wrong'}">${esc(p.text)||'∅ empty'}</div><span class="small">${p.errors.deletion}D · ${p.errors.insertion}I · ${p.errors.substitution}S</span></td></tr>`}).join('')}</tbody></table></section>
<section><div class="caption">Source-menu context · detector/crop outlines</div><a href="images/${r.id}_context.jpg" target="_blank"><img class="context" src="images/${r.id}_context.jpg" loading="lazy" alt="Context around crop"></a><p class="small">Geometry review flag: ${r.geometry_review_required?'YES':'no'} · overlap IDs: ${esc(r.overlap_ids.join(', ')||'none')} · detector confidence ${r.confidence}</p><a target="_blank" href="images/${r.menu}.jpg">Open full menu ${r.menu}</a><details><summary>Review guide</summary><p class="small">1. Is the opening mark really present in the source?<br>2. Does it survive into the upright crop and H64 input?<br>3. Is it touching an edge or much fainter/smaller than nearby glyphs?<br>4. If clearly visible, classify as a recognition miss. Choose “uncertain” when the VLM label cannot be verified.</p></details></section></div>
<div class="review"><select data-id="${r.id}" data-field="cause">${[['','Choose likely cause'],['recognition_miss','Visible bracket: recognition miss'],['clipped_crop','Clipped by crop boundary'],['small_thin','Very small / thin bracket'],['low_contrast','Low contrast / background'],['wrong_label','VLM label is wrong'],['ambiguous','Uncertain / ambiguous'],['correct','Prediction acceptable']].map(([v,t])=>`<option value="${v}" ${n.cause===v?'selected':''}>${t}</option>`).join('')}</select><textarea data-id="${r.id}" data-field="note" placeholder="Review note or corrected transcription">${esc(n.note||'')}</textarea></div></article>`}).join('')||'<p>No matching lines.</p>';
document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',()=>{notes[el.dataset.id]={...notes[el.dataset.id],[el.dataset.field]:el.value};save();$('count').textContent=`${rr.length} lines · ${Object.keys(notes).length} saved reviews`;}));}
['split','mode','reviewFilter'].forEach(id=>$(id).addEventListener('change',render));$('search').addEventListener('input',render);$('export').onclick=()=>{const blob=new Blob([JSON.stringify({version:'v50_opening_parentheses',created:new Date().toISOString(),review:notes},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='v50_parentheses_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();</script></html>'''


if __name__=='__main__': main()
