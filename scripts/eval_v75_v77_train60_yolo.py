"""Evaluate V75 (epoch_04 raw) and V77 (treatment step_0500 ema) on train_60 H=64 YOLO crops."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(r"D:\profolio\japanese_translation")
V74_DIR = ROOT / "outputs" / "recognition" / "experiments" / "v74_yolo25_v73_fullpasses2_colab"
V75_DIR = ROOT / "outputs" / "recognition" / "experiments" / "v75_v74_new_clean_yolo_fullpasses5_colab"
V76_DIR = ROOT / "outputs" / "recognition" / "experiments" / "v76_v75_dynamic_spacing_fullpasses5_colab"
V77_DIR = ROOT / "outputs" / "recognition" / "experiments" / "v77_v75_tight_gap_matched500_colab"

for p in (str(ROOT), str(V74_DIR / "code"), str(V75_DIR / "code"), str(V76_DIR / "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

from canonical_ctc import fold_parenthesis_logits
from data import IMAGENET_MEAN, IMAGENET_STD, output_length
from evaluate_v65 import load_v60_model
from post_completion_audit import levenshtein_dis, summarise_dis
from src.recognition.long_line_inference import ctc_tokens

TRAIN60_ROOT = ROOT / "data" / "raw" / "gemini_banana_mobile_menus_train_60"
LABELED_JSONL = TRAIN60_ROOT / "yolo_crops" / "labeled_crops.jsonl"
OUT_DIR = ROOT / "outputs" / "recognition" / "evaluation" / "v75_v77_train60_yolo"


def canonicalize(text: str) -> str:
    return text.replace("（", "(").replace("）", ")").strip()


@torch.inference_mode()
def eval_model_on_crops(
    model: torch.nn.Module,
    items: list[dict],
    char_to_id: dict[str, int],
    id_to_char: dict[int, str],
    device: torch.device,
) -> list[dict]:
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    records = []
    for item in items:
        with Image.open(item["ctc_path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        arr_f = (arr.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(arr_f.transpose(2, 0, 1)).unsqueeze(0).to(device)
        ori_t = torch.tensor([float(item["orientation_flag"])], dtype=torch.float32, device=device)
        vt = int(item["valid_T"])
        in_len = torch.tensor([vt], dtype=torch.long)

        logits = model(tensor, ori_t, in_len)
        logits = fold_parenthesis_logits(logits, char_to_id, drop_aliases=False)
        path = logits[:vt, 0].argmax(dim=-1).cpu().tolist()
        toks = [c for c, _, _ in ctc_tokens(path)]
        hyp = "".join(id_to_char.get(int(t), "") for t in toks)
        hyp_canon = canonicalize(hyp)
        ref_canon = item["canonical_text"]
        edits, d, ins, s = levenshtein_dis(ref_canon, hyp_canon)
        records.append({
            "crop_id": item["crop_id"],
            "source_menu": item["source_menu"],
            "orientation_flag": item["orientation_flag"],
            "tilt_group": item["tilt_group"],
            "dark_ambience": item["dark_ambience"],
            "is_split_window": item["is_split_window"],
            "valid_width": item["valid_width"],
            "ref": ref_canon,
            "pred": hyp_canon,
            "ref_len": len(ref_canon),
            "edits": edits,
            "D": d,
            "I": ins,
            "S": s,
            "exact": (hyp_canon == ref_canon),
        })
    return records


def breakdown(recs: list[dict]) -> dict:
    return {
        "all": summarise_dis(recs),
        "H_0": summarise_dis([r for r in recs if r["orientation_flag"] == 0]),
        "V_1": summarise_dis([r for r in recs if r["orientation_flag"] == 1]),
        "low_tilt_max4": summarise_dis([r for r in recs if r["tilt_group"] == "low_tilt_max4"]),
        "tilted_max15": summarise_dis([r for r in recs if r["tilt_group"] == "tilted_max15"]),
        "dark_ambience_true": summarise_dis([r for r in recs if r["dark_ambience"]]),
        "dark_ambience_false": summarise_dis([r for r in recs if not r["dark_ambience"]]),
        "single_window": summarise_dis([r for r in recs if not r["is_split_window"]]),
        "split_window_gt512": summarise_dis([r for r in recs if r["is_split_window"]]),
    }


def paired_menu_bootstrap(recs_a: list[dict], recs_b: list[dict], n_boot: int = 2000, seed: int = 20261115) -> dict:
    rng = np.random.default_rng(seed)
    by_menu = {}
    for ra, rb in zip(recs_a, recs_b):
        assert ra["crop_id"] == rb["crop_id"]
        m = ra["source_menu"]
        e = by_menu.setdefault(m, [0, 0, 0])
        e[0] += ra["edits"]
        e[1] += rb["edits"]
        e[2] += ra["ref_len"]
    menus = sorted(by_menu.keys())
    arr = np.array([by_menu[m] for m in menus], dtype=np.float64)
    deltas = []
    n = len(menus)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = arr[idx].sum(axis=0)
        cer_a = sub[0] / max(1.0, sub[2])
        cer_b = sub[1] / max(1.0, sub[2])
        deltas.append(cer_b - cer_a)
    deltas = np.array(deltas)
    return {
        "num_menus": n,
        "mean_delta_v77_minus_v75": round(float(deltas.mean()), 6),
        "ci95": [round(float(np.percentile(deltas, 2.5)), 6), round(float(np.percentile(deltas, 97.5)), 6)],
        "p_v77_better": round(float((deltas < 0).mean()), 4),
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    menu_meta = {}
    for line in (TRAIN60_ROOT / "prompts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            menu_meta[f"menu-{int(row['id']):03d}"] = {
                "tilt_group": row["tilt_group"],
                "dark_ambience": bool(row["dark_ambience"]),
            }

    items = []
    for line in LABELED_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "OK":
            continue
        text = canonicalize(r["text"])
        if not text:
            continue
        mm = menu_meta[r["source_menu"]]
        vw = int(r["valid_width"])
        items.append({
            "crop_id": r["crop_id"],
            "source_menu": r["source_menu"],
            "orientation_flag": int(r["orientation_flag"]),
            "tilt_group": mm["tilt_group"],
            "dark_ambience": mm["dark_ambience"],
            "is_split_window": ("__w" in r["crop_id"]),
            "valid_width": vw,
            "valid_T": int(r.get("input_length") or output_length(vw)),
            "ctc_path": TRAIN60_ROOT / "yolo_crops" / r["ctc_crop"],
            "canonical_text": text,
        })

    vocab = json.loads((V74_DIR / "payload" / "vocab.json").read_text(encoding="utf-8"))
    char_to_id = vocab["char_to_id"]
    id_to_char = {int(v): k for k, v in char_to_id.items()}
    classes = len(char_to_id) + 1

    v75_ckpt = V75_DIR / "downloads" / "epoch_04.pt"
    v77_ckpt = V77_DIR / "downloads" / "treatment" / "step_0500.pt"

    m75 = load_v60_model(v75_ckpt, "raw", classes, device)
    m75.eval()
    recs_75 = eval_model_on_crops(m75, items, char_to_id, id_to_char, device)

    m77 = load_v60_model(v77_ckpt, "ema", classes, device)
    m77.eval()
    recs_77 = eval_model_on_crops(m77, items, char_to_id, id_to_char, device)

    (OUT_DIR / "predictions_v75.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs_75) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "predictions_v77.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs_77) + "\n", encoding="utf-8"
    )

    summary = {
        "dataset": str(LABELED_JSONL),
        "evaluated_ok_crops": len(items),
        "v75_ckpt": str(v75_ckpt),
        "v77_ckpt": str(v77_ckpt),
        "v75": breakdown(recs_75),
        "v77": breakdown(recs_77),
        "paired_bootstrap_v77_vs_v75": paired_menu_bootstrap(recs_75, recs_77),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
