"""Local webtool: upload a menu image -> run V5 YOLO11n-OBB crop logic -> run l2_v02 ep04_raw -> inspect top-5 logits per timestep."""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_crop_canvas_pipeline import (
    DEFAULT_WEIGHTS,
    LOCAL_PKGS,
    RUNTIME_ROOT,
    canonicalize_obb_corners,
    expand_quad,
    sort_detections_reading_order,
    warp_polygon,
)
from src.detection.line_crops import process as process_lines, recognition_input
from src.recognition.canonical_ctc import fold_parenthesis_logits
from src.recognition.data import IMAGENET_MEAN, IMAGENET_STD, decode, output_length
from src.recognition.model import ARCH_SPECS, MenuRecognizer

L2_V03_EXP = ROOT / "outputs" / "recognition" / "experiments" / "h4_k41_l2_v03_yolo_all240"
DEFAULT_OCR_CKPT = L2_V03_EXP / "downloads" / "h4_k41_l2_v03" / "epoch_05.pt"
DEFAULT_VOCAB = L2_V03_EXP / "payload" / "vocab.json"


def _img_to_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class InferenceEngine:
    """Holds YOLO11n-OBB detector and l2_v02_ep04_raw MenuRecognizer."""

    def __init__(self, yolo_weights: Path, ocr_ckpt: Path, vocab_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_ROOT))
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        if LOCAL_PKGS.is_dir() and str(LOCAL_PKGS) not in sys.path:
            sys.path.append(str(LOCAL_PKGS))
        from ultralytics import YOLO

        self.yolo = YOLO(str(yolo_weights))
        self.yolo_weights = yolo_weights

        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.char_to_id: dict[str, int] = vocab["char_to_id"]
        self.id_to_char: dict[int, str] = {int(v): k for k, v in self.char_to_id.items()}
        self.id_to_char[0] = "<blank>"
        classes = len(vocab["characters"]) + 1

        spec = ARCH_SPECS["h4_k41"]
        ck = torch.load(ocr_ckpt, map_location="cpu", weights_only=False)
        self.ocr = MenuRecognizer(
            classes=classes,
            pretrained=False,
            temporal=True,
            orientation=True,
            temporal_norm="batch",
            use_bilstm=True,
            height_pool=spec["height_pool"],
            stage_strides=spec["stage_strides"],
            expected_h=spec["expected_h"],
            bilstm_layers=spec["bilstm_layers"],
        ).eval()
        self.ocr.load_state_dict(ck["raw"], strict=True)
        self.ocr.to(self.device)
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    @torch.inference_mode()
    def predict_image(self, image_bytes: bytes, confidence: float = 0.35, imgsz: int = 640) -> dict:
        rgb_im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = rgb_im.size

        # 1. Run exact V5 YOLO11n-OBB crop logic from scripts/yolo_crop_canvas_pipeline.py
        t_yolo0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_path = Path(tf.name)
            rgb_im.save(tmp_path)
        try:
            result = self.yolo.predict(
                str(tmp_path),
                imgsz=imgsz,
                conf=confidence,
                iou=0.7,
                max_det=300,
                device=str(self.device),
                half=False,
                augment=False,
                verbose=False,
            )[0]
        finally:
            tmp_path.unlink(missing_ok=True)
        yolo_ms = (time.perf_counter() - t_yolo0) * 1000.0

        pad_x, pad_y = 0.04, 0.06
        iou_dedup, merge_iom = 0.65, 0.20
        raw_dets = []
        if result.obb is not None:
            for points, cls_id, score in zip(
                result.obb.xyxyxyxy.cpu().numpy(),
                result.obb.cls.cpu().numpy(),
                result.obb.conf.cpu().numpy(),
            ):
                label = str(self.yolo.names[int(cls_id)])
                quad, box_w, box_h, vert = canonicalize_obb_corners(points, label=label)
                padded_quad, pad_w, pad_h = expand_quad(quad, pad_x, pad_y, img_w, img_h)
                raw_dets.append({
                    "label": label,
                    "class_id": int(cls_id),
                    "confidence": float(score),
                    "obb_polygon": quad.tolist(),
                    "padded_polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in padded_quad.tolist()],
                    "box_size": [round(pad_w, 1), round(pad_h, 1)],
                    "center": quad.mean(axis=0).tolist(),
                    "is_vertical": vert,
                })

        processed = process_lines(
            [
                dict(id=str(i), points=d["obb_polygon"], label=d["label"], confidence=d["confidence"])
                for i, d in enumerate(raw_dets)
            ],
            img_w,
            img_h,
            duplicate_iou=iou_dedup,
            min_merge_iom=merge_iom,
        )
        deduped_dets = []
        for item in processed["crops"]:
            vert = bool(item["orientation_flag"])
            core = np.asarray(item["core"])
            q = np.asarray(item["crop_polygon"])
            if vert:
                core = np.roll(core, 1, axis=0)
                q = np.roll(q, 1, axis=0)
            w = float((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2)
            h = float((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)
            deduped_dets.append(dict(
                label=item["label"],
                class_id=1 if vert else 0,
                confidence=item["confidence"],
                obb_polygon=core.tolist(),
                padded_polygon=q.tolist(),
                box_size=[w, h],
                center=core.mean(0).tolist(),
                is_vertical=vert,
            ))
        ordered_dets = sort_detections_reading_order(deduped_dets)

        # Draw annotated preview
        annotated = rgb_im.copy()
        draw = ImageDraw.Draw(annotated)
        crops_out = []
        t_ocr0 = time.perf_counter()

        for line_idx, det in enumerate(ordered_dets, start=1):
            padded_quad = np.asarray(det["padded_polygon"], dtype=float)
            pad_w, pad_h = det["box_size"]
            display_crop_im = warp_polygon(rgb_im, padded_quad, (pad_w, pad_h))
            strip = display_crop_im.transpose(Image.Transpose.ROTATE_90) if det["is_vertical"] else display_crop_im
            ctc_crop_im, norm_meta = recognition_input(strip)

            valid_width = int(norm_meta["valid_width"])
            valid_t = max(1, min(output_length(valid_width), output_length(ctc_crop_im.width)))
            arr = np.asarray(ctc_crop_im.convert("RGB"), dtype=np.uint8)
            arr_f = (arr.astype(np.float32) / 255.0 - self.mean) / self.std
            tensor = torch.from_numpy(arr_f.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            ori_flag = 1.0 if det["is_vertical"] else 0.0
            ori_t = torch.tensor([ori_flag], dtype=torch.float32, device=self.device)
            in_len = torch.tensor([valid_t], dtype=torch.long)

            logits = self.ocr(tensor, ori_t, in_len)
            logits = fold_parenthesis_logits(logits, self.char_to_id, drop_aliases=False)
            seq_logits = logits[:valid_t, 0, :].float()
            pred_text = decode(logits[:valid_t], self.char_to_id, in_len)[0]

            probs = torch.softmax(seq_logits, dim=-1)
            top5_vals, top5_ids = torch.topk(seq_logits, k=5, dim=-1)
            top5_probs = torch.gather(probs, dim=-1, index=top5_ids)

            timesteps = []
            for t in range(valid_t):
                entries = []
                for rank in range(5):
                    tid = int(top5_ids[t, rank].item())
                    entries.append({
                        "id": tid,
                        "token": self.id_to_char.get(tid, f"#{tid}"),
                        "logit": round(float(top5_vals[t, rank].item()), 3),
                        "prob": round(float(top5_probs[t, rank].item()), 4),
                    })
                timesteps.append({"t": t, "top5": entries})

            color = "#38bdf8" if det["is_vertical"] else "#34d399"
            poly_pts = [(float(x), float(y)) for x, y in padded_quad]
            draw.polygon(poly_pts, outline=color, width=3)
            draw.text((poly_pts[0][0] + 4, max(2.0, poly_pts[0][1] - 16)), f"#{line_idx}", fill=color)

            crops_out.append({
                "index": line_idx,
                "orientation": "V" if det["is_vertical"] else "H",
                "yolo_label": det["label"],
                "confidence": round(float(det["confidence"]), 3),
                "valid_width": valid_width,
                "valid_T": valid_t,
                "prediction": pred_text,
                "display_crop_url": _img_to_data_url(display_crop_im),
                "ctc_crop_url": _img_to_data_url(ctc_crop_im),
                "timesteps": timesteps,
            })

        ocr_ms = (time.perf_counter() - t_ocr0) * 1000.0
        return {
            "image_size": [img_w, img_h],
            "yolo_ms": round(yolo_ms, 1),
            "ocr_ms": round(ocr_ms, 1),
            "num_crops": len(crops_out),
            "annotated_url": _img_to_data_url(annotated),
            "crops": crops_out,
        }


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Menu YOLO + L2_v03 Inspector</title>
<style>
  :root {
    --bg: #0f172a; --panel: #1e293b; --panel2: #0b1220; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8; --green: #34d399; --amber: #fbbf24;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); }
  header { padding: 14px 22px; border-bottom: 1px solid var(--border); display: flex; flex-wrap: wrap; align-items: center; gap: 16px; background: #0b1220; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  .badge { font-size: 12px; padding: 3px 8px; border-radius: 999px; background: #1e293b; border: 1px solid var(--border); color: var(--muted); }
  main { max-width: 1500px; margin: 0 auto; padding: 18px 22px; display: grid; gap: 18px; }
  .controls { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }
  input[type=file] { color: var(--text); }
  button { background: var(--accent); color: #082f49; border: 0; border-radius: 8px; padding: 9px 16px; font-weight: 700; cursor: pointer; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .preview-wrap { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; display: none; }
  .preview-wrap img { max-width: 100%; max-height: 540px; border-radius: 6px; display: block; margin: 0 auto; }
  .crops { display: grid; gap: 16px; }
  .crop-card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; display: grid; gap: 12px; }
  .crop-head { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
  .pred-text { font-size: 22px; font-weight: 700; color: var(--green); letter-spacing: 0.03em; }
  .ori-H { color: var(--green); border-color: var(--green); }
  .ori-V { color: var(--accent); border-color: var(--accent); }
  .crop-imgs { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  .crop-imgs figure { margin: 0; background: var(--panel2); border: 1px solid var(--border); border-radius: 6px; padding: 6px; }
  .crop-imgs figcaption { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .crop-imgs img { max-height: 96px; max-width: 100%; display: block; }
  .ts-scroll { display: flex; gap: 8px; overflow-x: auto; padding: 6px 2px 10px; }
  .ts-col { min-width: 124px; background: var(--panel2); border: 1px solid var(--border); border-radius: 8px; padding: 7px; flex-shrink: 0; }
  .ts-col.nonblank { border-color: var(--amber); }
  .ts-hdr { font-size: 11px; font-weight: 700; color: var(--muted); margin-bottom: 6px; display: flex; justify-content: space-between; }
  .tok-row { display: grid; grid-template-columns: 28px 1fr auto; gap: 4px; align-items: center; font-size: 12px; padding: 2px 0; }
  .tok-sym { font-weight: 700; text-align: center; background: #1e293b; border-radius: 4px; padding: 1px 3px; }
  .tok-sym.blank { color: var(--muted); font-weight: 400; font-size: 10px; }
  .tok-logit { font-family: ui-monospace, Consolas, monospace; font-size: 11px; color: var(--text); text-align: right; }
  .tok-prob { font-family: ui-monospace, Consolas, monospace; font-size: 10px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Menu YOLO11n-OBB + L2_v03 (ep05_raw) Inspector</h1>
  <span class="badge">YOLO V5 OBB + line_crops</span>
  <span class="badge">h4_k41 l2_v03 epoch_05.pt (raw)</span>
  <span id="status" class="badge">Ready</span>
</header>
<main>
  <div class="controls">
    <input id="fileInput" type="file" accept="image/*" />
    <label>Conf <input id="confInput" type="number" min="0.05" max="0.95" step="0.05" value="0.35" style="width:64px"></label>
    <button id="runBtn">Run YOLO + OCR</button>
    <span id="timing" class="badge" style="display:none"></span>
  </div>
  <div id="previewWrap" class="preview-wrap">
    <img id="annotatedImg" alt="Annotated detections" />
  </div>
  <div id="crops" class="crops"></div>
</main>
<script>
const fileInput = document.getElementById('fileInput');
const confInput = document.getElementById('confInput');
const runBtn = document.getElementById('runBtn');
const statusEl = document.getElementById('status');
const timingEl = document.getElementById('timing');
const previewWrap = document.getElementById('previewWrap');
const annotatedImg = document.getElementById('annotatedImg');
const cropsEl = document.getElementById('crops');

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

runBtn.addEventListener('click', async () => {
  const f = fileInput.files[0];
  if (!f) { alert('Please select an image first.'); return; }
  runBtn.disabled = true;
  statusEl.textContent = 'Running YOLO + L2_v02...';
  cropsEl.innerHTML = '';
  try {
    const b64 = await new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = () => res(r.result.split(',')[1]);
      r.onerror = rej;
      r.readAsDataURL(f);
    });
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ image_b64: b64, confidence: parseFloat(confInput.value || '0.35') }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Request failed');
    annotatedImg.src = data.annotated_url;
    previewWrap.style.display = 'block';
    timingEl.style.display = 'inline-block';
    timingEl.textContent = `${data.num_crops} crops | YOLO: ${data.yolo_ms}ms | OCR: ${data.ocr_ms}ms`;
    statusEl.textContent = 'Done';

    for (const c of data.crops) {
      const card = document.createElement('div');
      card.className = 'crop-card';
      const tsHtml = c.timesteps.map(ts => {
        const top1 = ts.top5[0];
        const nb = top1.id !== 0 ? 'nonblank' : '';
        const rows = ts.top5.map(e => {
          const isBlank = e.id === 0;
          const lbl = isBlank ? 'blk' : esc(e.token);
          return `<div class="tok-row">
            <span class="tok-sym ${isBlank ? 'blank' : ''}">${lbl}</span>
            <span class="tok-logit">${e.logit.toFixed(2)}</span>
            <span class="tok-prob">${(e.prob * 100).toFixed(1)}%</span>
          </div>`;
        }).join('');
        return `<div class="ts-col ${nb}"><div class="ts-hdr"><span>t=${ts.t}</span><span>${top1.id === 0 ? 'blk' : esc(top1.token)}</span></div>${rows}</div>`;
      }).join('');

      card.innerHTML = `
        <div class="crop-head">
          <span class="badge">#${c.index}</span>
          <span class="badge ori-${c.orientation}">${c.orientation === 'V' ? 'Vertical (V)' : 'Horizontal (H)'}</span>
          <span class="badge">${esc(c.yolo_label)} (${c.confidence})</span>
          <span class="badge">w=${c.valid_width}px | T=${c.valid_T}</span>
          <span class="pred-text">${esc(c.prediction || '(empty)')}</span>
        </div>
        <div class="crop-imgs">
          <figure><figcaption>Rectified Crop</figcaption><img src="${c.display_crop_url}" /></figure>
          <figure><figcaption>CTC Input (H=64)</figcaption><img src="${c.ctc_crop_url}" /></figure>
        </div>
        <div class="ts-scroll">${tsHtml}</div>
      `;
      cropsEl.appendChild(card);
    }
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  } finally {
    runBtn.disabled = false;
  }
});
</script>
</body>
</html>
"""


def make_handler(engine: InferenceEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"Not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/predict":
                self._send(404, b'{"error":"Not found"}', "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                img_bytes = base64.b64decode(payload["image_b64"])
                conf = float(payload.get("confidence", 0.35))
                out = engine.predict_image(img_bytes, confidence=conf)
                self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            except Exception as exc:
                err = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(500, err, "application/json; charset=utf-8")

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--yolo-weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--ocr-ckpt", type=Path, default=DEFAULT_OCR_CKPT)
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    args = p.parse_args()

    engine = InferenceEngine(args.yolo_weights, args.ocr_ckpt, args.vocab)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"SERVE_READY http://{args.host}:{args.port}/ (device={engine.device})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
