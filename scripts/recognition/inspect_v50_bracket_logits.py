"""Forward-only per-frame opening-parenthesis probability/rank diagnostics."""
from collections import defaultdict
from pathlib import Path
import html
import json
import math
import numpy as np
from PIL import Image
import torch
from src.recognition.model import MenuRecognizer
from src.recognition.data import preprocess, collate, decode
from src.recognition.canonical_ctc import fold_parenthesis_logits
from src.recognition.utils import sha, write_json

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/recognition/pilots/v50_normalized_matched_2000'
OUT=ROOT/'outputs/recognition/review/v50_opening_parentheses'
INPUT=ROOT/'data/processed/gemini_banana_mobile_menus_50_crops'


def main():
    torch.set_num_threads(4); assert torch.cuda.is_available()
    review=json.loads((OUT/'index.json').read_text(encoding='utf-8'))
    targets={r['id']:r for r in review['rows'] if r['split']=='val_holdout'}
    labels=[json.loads(s) for s in (ROOT/'outputs/recognition/training/v40_new_characters_2000/yolo_labels.jsonl').read_text(encoding='utf-8').splitlines()]
    labels=[r for r in labels if r['split']=='val_holdout'];assert len(targets)==13 and len(labels)==156
    checkpoints={'V43':ROOT/'checkpoints/menu_recognizer_v43.pt',
        'Control selected 1500':RUN/'control/best.pt',
        'Control final 2000':RUN/'control/final_raw.pt',
        'BiLSTM final 2000':RUN/'bilstm/final_raw.pt'}
    results=[]; arrays={}; artifacts={}
    for tag,path in checkpoints.items():
        saved=torch.load(path,map_location='cpu',weights_only=False);mapping=saved['vocab']['char_to_id']
        assert all(c in mapping for c in '()（）');cfg=saved['config']
        model=MenuRecognizer(len(mapping)+1,use_bilstm=True).cuda().eval()
        model.load_state_dict(saved['model'],strict=True); del saved
        groups=defaultdict(list)
        for i,r in enumerate(labels):groups[r['canvas']].append(i)
        reverse={v:k for k,v in mapping.items()}; reverse[0]='[blank]'
        parity=[]
        with torch.inference_mode():
            for canvas,ids in sorted(groups.items()):
                for start in range(0,len(ids),32):
                    indices=ids[start:start+32]; samples=[]
                    for i in indices:
                        r=labels[i];assert sha(INPUT/r['image'])==r['sha256']
                        with Image.open(INPUT/r['image']) as im:t,w,compressed=preprocess(im,cfg['mean'],cfg['std'],'batch',max_width=canvas)
                        assert not compressed and w==canvas
                        samples.append((t,torch.tensor([mapping[c] for c in r['text']]),r['orientation_flag'],r['valid_width'],r['text'],False))
                    batch=collate(samples,padded_width=canvas)
                    with torch.autocast('cuda',dtype=torch.float16):
                        z=model(batch['images'].cuda(),batch['orientation'].cuda(),batch['input_lengths'])
                    pooled=fold_parenthesis_logits(z,mapping)
                    raw=decode(z,mapping,batch['input_lengths']);canonical=decode(pooled,mapping,batch['input_lengths']);full=decode(pooled,mapping)
                    for j,i in enumerate(indices):
                        meta=labels[i];ident=meta['crop_id']
                        if ident not in targets:continue
                        reference=targets[ident]['predictions'][tag]
                        assert raw[j]==reference['validation']['text'],(tag,ident,'raw parity',raw[j])
                        assert canonical[j]==reference['canonical_validation']['text'],(tag,ident,'canonical parity')
                        parity.append(ident);valid=int(batch['input_lengths'][j]); total=z.shape[0]
                        rawlogits=z[:,j].float().cpu();logits=pooled[:,j].double().cpu()
                        lp=logits.log_softmax(-1);prob=lp.exp();open_id=mapping['('];close_id=mapping[')']
                        rank=(logits>logits[:,open_id,None]).sum(-1)+1
                        peak=int(prob[:valid,open_id].argmax());best_rank=int(rank[:valid].min())
                        competing=logits.clone();competing[:,open_id]=float('-inf')
                        boost=(competing.max(-1).values-logits[:,open_id]); best_boost=float(boost[:valid].min())
                        frames=[]
                        for t in range(total):
                            pp,ii=prob[t].topk(5)
                            frames.append(dict(frame=t,valid=t<valid,x_stride4=t*4,
                                p_open=float(prob[t,open_id]),p_close=float(prob[t,close_id]),p_blank=float(prob[t,0]),
                                logp_open=float(lp[t,open_id]),open_rank=int(rank[t]),
                                top5=[dict(token=reverse[int(k)],p=float(p)) for p,k in zip(pp,ii)]))
                        literal={c:dict(max_probability=float(rawlogits[:valid].double().softmax(-1)[:,mapping[c]].max()),
                            finite_logits=bool(torch.isfinite(rawlogits[:,mapping[c]]).all())) for c in '()（）'}
                        result=dict(id=ident,model=tag,label=meta['text'],raw=raw[j],canonical=canonical[j],
                            full_width_decode=full[j],valid_frames=valid,physical_frames=total,
                            peak_frame=peak,peak_p_open=float(prob[peak,open_id]),peak_open_rank=int(rank[peak]),
                            peak_p_blank=float(prob[peak,0]),peak_top5=frames[peak]['top5'],
                            best_open_rank=best_rank,top5_frame_count=int((rank[:valid]<=5).sum()),
                            top10_frame_count=int((rank[:valid]<=10).sum()),
                            min_logit_boost_to_win_any_valid_frame=best_boost,
                            min_logp_open=float(lp[:valid,open_id].min()),
                            zero_p_open_float32_frames=int((prob[:valid,open_id].float()==0).sum()),
                            zero_p_open_float64_frames=int((prob[:valid,open_id]==0).sum()),
                            padded_tail_open_argmax_frames=int((logits[valid:].argmax(-1)==open_id).sum()),
                            literal_width_variants=literal,frames=frames)
                        assert result['zero_p_open_float64_frames']==0 and all(v['finite_logits'] for v in literal.values())
                        results.append(result);arrays[tag.replace(' ','_')+'__'+ident]=rawlogits.numpy()
        assert len(parity)==13
        artifacts[tag]=dict(checkpoint=str(path.relative_to(ROOT)),sha256=sha(path),saved_fp16_predictions_matched=13,
            vocabulary_ids={c:mapping[c] for c in '()（）'})
        del model;torch.cuda.empty_cache()
    np.savez_compressed(OUT/'raw_logits_fp16_forward.npz',**arrays)
    report=dict(precision='FP16 forward; probability/log-softmax recomputed in float64 from finite logits',
        semantic_open='P(() + P(（), pooled before CTC collapse',checkpoints=artifacts,
        reviewed_holdout_only=True,training_performed=False,results=results)
    write_json(OUT/'logits_analysis.json',report)
    build_page(report)
    for tag in checkpoints:
        rr=[r for r in results if r['model']==tag]
        print(tag,json.dumps(dict(min_peak=min(r['peak_p_open'] for r in rr),max_peak=max(r['peak_p_open'] for r in rr),
            lines_with_top5=sum(r['top5_frame_count']>0 for r in rr),lines_with_top10=sum(r['top10_frame_count']>0 for r in rr),
            lines_best_rank1=sum(r['best_open_rank']==1 for r in rr),zero_p_frames=sum(r['zero_p_open_float64_frames'] for r in rr),
            tail_open_winners=sum(r['padded_tail_open_argmax_frames'] for r in rr))))


def build_page(report):
    esc=html.escape;groups=defaultdict(list)
    for r in report['results']:groups[r['id']].append(r)
    chunks=['''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parenthesis logits</title>
<style>body{font:16px system-ui;background:#111922;color:#e3eaf2;margin:30px}a{color:#80c8ff}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #415268}section{background:#1b2837;padding:20px;margin:25px 0;border-radius:10px}img{max-width:100%;max-height:110px;background:white}summary{cursor:pointer;margin:12px 0}.scroll{max-height:450px;overflow:auto}p{line-height:1.6}small{color:#bac8d7}</style>
<h1>Opening-parenthesis logits · 13 reviewed YOLO failures</h1><a href="/">Back to crop review</a>
<p>FP16 forward matches all saved predictions. Probabilities below use float64 softmax, pooling ASCII/full-width opening brackets. All probabilities are positive; no opening bracket wins a valid frame. “Top 5” means it is present among plausible alternatives, not that greedy decoding emits it. Peak rank is measured at maximum opening probability; best rank may occur at another frame. The required boost beats every competitor, not only blank. Frame x≈4t is stride position, not an exact glyph boundary.</p>''']
    for ident,rr in groups.items():
        chunks.append(f'<section><h2>{esc(ident)} · {esc(rr[0]["label"])}</h2><img src="images/{ident}_native.png"><div class="scroll"><table><tr><th>Model</th><th>Peak P(open)</th><th>P(blank) there</th><th>Peak / best rank</th><th>Valid top5 frames</th><th>Min boost to win</th></tr>')
        for r in rr:
            chunks.append(f'<tr><td>{esc(r["model"])}</td><td>{100*r["peak_p_open"]:.5g}%</td><td>{100*r["peak_p_blank"]:.4g}%</td><td>{r["peak_open_rank"]} / {r["best_open_rank"]}</td><td>{r["top5_frame_count"]}/{r["valid_frames"]}</td><td>{r["min_logit_boost_to_win_any_valid_frame"]:.3f}</td></tr>')
        chunks.append('</table></div>')
        for r in rr:
            chunks.append(f'<details><summary>{esc(r["model"])}: every frame and top five tokens</summary><p>Decoded: {esc(r["canonical"])}<br>Full physical width: {esc(r["full_width_decode"])}<br>Peak frame: {r["peak_frame"]}; valid T={r["valid_frames"]}; physical T={r["physical_frames"]}</p><div class="scroll"><table><tr><th>Frame</th><th>Valid?</th><th>P(open)</th><th>P(blank)</th><th>Open rank</th><th>Top 5</th></tr>')
            for f in r['frames']:
                top=' · '.join(f'{esc(k["token"])} {100*k["p"]:.3g}%' for k in f['top5'])
                chunks.append(f'<tr><td>{f["frame"]}</td><td>{f["valid"]}</td><td>{100*f["p_open"]:.5g}%</td><td>{100*f["p_blank"]:.4g}%</td><td>{f["open_rank"]}</td><td>{top}</td></tr>')
            chunks.append('</table></div></details>')
        chunks.append('</section>')
    (OUT/'logits.html').write_text('\n'.join(chunks),encoding='utf-8')


if __name__=='__main__':main()
