"""Generate 60 diverse prompt specs and JSON sidecars for mobile menu training images:
- 30 low-tilt (<=4 deg) + 30 tilted (at most 15 deg)
- 20% (12/60: 6 low-tilt + 6 tilted) darker ambience lighting
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(r"D:\profolio\japanese_translation")
if str(ROOT / "scripts" / "data") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "data"))

from build_train_menus_50_prompts import CATEGORIES, INTERIORS, TEXTURES, TYPOGRAPHIES

OUT_DIR = ROOT / "data" / "raw" / "gemini_banana_mobile_menus_train_60"

LOW_TILT_ANGLES = [
    "straight-on perpendicular overhead flat-lay shot, minimal tilt under 4 degrees",
    "nearly perpendicular eye-level handheld shot facing the menu directly, minimal tilt under 4 degrees",
    "centered front-facing mobile camera shot with negligible perspective skew (under 4 degrees)",
]

TILT_15_ANGLES = [
    "handheld three-quarter diner perspective from left with moderate tilt and roll up to 15 degrees",
    "angled table-side smartphone shot from lower right with perspective tilt between 10 and 15 degrees",
    "slightly oblique overhead angle with noticeable diagonal tilt not exceeding 15 degrees",
    "casual seated diner perspective tilted up to 15 degrees",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for idx in range(1, 61):
        cat_idx = (idx - 1) % len(CATEGORIES)
        cat_slug, cat_title, lines = CATEGORIES[cat_idx]
        pass_num = (idx - 1) // len(CATEGORIES)
        if pass_num == 1:
            variant_tag = " (Late Night Edition)"
            mod_lines = [lines[0]] + [
                l.replace("580円", "620円").replace("480円", "520円").replace("880円", "920円").replace("350円", "380円")
                for l in lines[1:]
            ]
        elif pass_num == 2:
            variant_tag = " (Seasonal Edition)"
            mod_lines = [lines[0]] + [
                l.replace("580円", "550円").replace("480円", "450円").replace("880円", "850円").replace("220円", "240円")
                for l in lines[1:]
            ]
        else:
            variant_tag = ""
            mod_lines = list(lines)

        texture = TEXTURES[(idx * 5 + 3) % len(TEXTURES)]
        typography = TYPOGRAPHIES[(idx * 7 + 1) % len(TYPOGRAPHIES)]
        interior = INTERIORS[(idx * 3 + 5) % len(INTERIORS)]

        if "chalkboard" in texture and "chalk" not in typography and "paint-marker" not in typography:
            typography = "hand-lettered white chalk script on dark slate surface"
        elif "chalk" in typography and "chalkboard" not in texture and "yakisugi" not in texture:
            texture = "matte black chalkboard menu with faint chalk dust smudges"
        elif "paint-marker" in typography and "wood" not in texture and "yakisugi" not in texture:
            texture = "weathered dark cedar wood board with natural wood grain"
        elif "cream paper" in typography and "wood" in texture:
            typography = "bold dark sumi brush lettering carved/painted cleanly onto the wooden slats"

        is_tilted_15 = idx > 30
        max_tilt = 15 if is_tilted_15 else 4
        angle = (
            TILT_15_ANGLES[(idx - 31) % len(TILT_15_ANGLES)]
            if is_tilted_15
            else LOW_TILT_ANGLES[(idx - 1) % len(LOW_TILT_ANGLES)]
        )
        tilt_clause = (
            "Camera tilt and roll between 8 and 15 degrees (must NOT exceed 15 degrees)."
            if is_tilted_15
            else "Camera tilt and roll must be minimal (NOT exceeding 4 degrees)."
        )

        # Exactly 20% (12/60: 6 low-tilt + 6 tilted) darker ambience lighting
        is_dark_ambience = (idx % 5 == 0)
        lighting_clause = (
            "Lighting: darker evening izakaya ambience, dim warm low-key tungsten lighting with soft natural shadows across the menu while keeping Japanese text legible."
            if is_dark_ambience
            else "Lighting: natural indoor restaurant ambient lighting with soft realistic shadows."
        )

        intended_text = " / ".join(mod_lines)
        quoted_lines = ", ".join(f'"{l}"' for l in mod_lines)
        prompt = (
            "Use case: photorealistic-natural. Asset: realistic mobile phone photograph of a Japanese restaurant menu for OCR training. "
            "Simulate an unedited smartphone photo (26mm focal length, f/1.8, natural indoor mobile camera processing) taken by a diner "
            "so the menu fills 80% to 90% of the frame, with 10% to 20% surrounding restaurant context "
            f"({interior}) softly visible around the outer border. "
            f"Camera angle: {angle}. {tilt_clause} "
            f"{lighting_clause} "
            f"Menu material and texture: {texture}. "
            f"Typography: {typography}. "
            "Render ONLY genuine, sharp, orthographically accurate Japanese characters and numerals "
            "(no fake pseudo-kanji strokes, no English translations, no romanization, no watermarks, no collage). "
            f"Japanese menu text to render faithfully on the menu: {quoted_lines}. "
            "Keep all text inside the menu boundaries with no clipped characters. High resolution."
        )

        item = {
            "id": idx,
            "category": cat_slug,
            "title": f"{cat_title}{variant_tag}",
            "status": "pending_image",
            "provider": "gemini_banana",
            "image": f"menu-{idx:03d}.jpg",
            "angle": angle,
            "tilt_group": "tilted_max15" if is_tilted_15 else "low_tilt_max4",
            "max_camera_tilt_deg": max_tilt,
            "dark_ambience": is_dark_ambience,
            "interior": interior,
            "texture": texture,
            "typography": typography,
            "intended_text": intended_text,
            "gt_lines": mod_lines,
            "prompt": prompt,
            "reviewed": False,
            "source": "genai_synthetic_mobile_menu_train_60",
            "transcription_verified": False,
        }
        items.append(item)
        (OUT_DIR / f"menu-{idx:03d}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (OUT_DIR / "prompts.jsonl").write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n", encoding="utf-8")
    manifest = {
        "collection_name": "gemini_banana_mobile_menus_train_60",
        "total_prompts": 60,
        "completed": 0,
        "low_tilt_count": sum(1 for it in items if not it["max_camera_tilt_deg"] > 4),
        "tilted_15_count": sum(1 for it in items if it["max_camera_tilt_deg"] == 15),
        "dark_ambience_count": sum(1 for it in items if it["dark_ambience"]),
        "prompt_sidecars": "menu-001.json through menu-060.json",
        "jsonl_pack": "prompts.jsonl",
        "items": items,
    }
    (OUT_DIR / "collection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("collection_name", "total_prompts", "low_tilt_count", "tilted_15_count", "dark_ambience_count")}, indent=2))


if __name__ == "__main__":
    main()
